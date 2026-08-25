"""Headless worker peer inbox: goal/loop workers receive over their socket.

The policy surface is narrower than the TUI's by necessity: nothing here can
answer a hold prompt, so holds degrade to explicit drops rather than parking
silently. These tests pin that behavior, the queue/drain split, and the
recorded PEER_MESSAGE_* events.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from kolega_code.cli.peer_messaging import HeadlessPeerInbox, start_headless_peer_inbox
from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import CliSettings, SettingsStore
from kolega_code.events import KnownEventType
from kolega_code.session.inbox import MESSAGING_ENV_FLAG, send_over_socket


class _FakeAgent:
    def __init__(self) -> None:
        self.messaging_socket_path: Path | None = None
        self.provider: Any = None

    def set_queued_input_provider(self, provider: Any) -> None:
        self.provider = provider


_STATE_ROOTS: list[Path] = []


def _short_state_root() -> Path:
    try:
        # /tmp explicitly: pytest rewrites TMPDIR into its deep per-test tree,
        # which can exceed the 104-byte AF_UNIX limit on macOS.
        root = Path(tempfile.mkdtemp(prefix="kolega-head-", dir="/tmp"))
    except OSError:
        root = Path(tempfile.mkdtemp(prefix="kolega-head-"))
    _STATE_ROOTS.append(root)
    return root


@pytest.fixture(autouse=True)
def _cleanup_state_roots():
    yield
    import shutil

    while _STATE_ROOTS:
        shutil.rmtree(_STATE_ROOTS.pop(), ignore_errors=True)


def _inbox(state_root: Path, *, settings: CliSettings | None = None, mode: str = "ask"):
    project = state_root / "project"
    project.mkdir(exist_ok=True)
    store = SessionStore(state_root)
    if settings is not None:
        SettingsStore(state_root).save(settings)
    session = store.create(project, "code", {"model": "test"}, title="worker")
    inbox = HeadlessPeerInbox(
        store_root=state_root,
        session_id=session.session_id,
        settings=settings or CliSettings(),
        permission_mode_value=mode,
        json_mode=True,
        journal_factory=lambda: FileSessionEventStore(store.journal(session.session_id)),
    )
    return inbox, store, session


@pytest.mark.asyncio
async def test_start_binds_socket_and_wires_the_provider() -> None:
    state_root = _short_state_root()
    try:
        inbox, _store, _session = _inbox(state_root)
        agent = _FakeAgent()

        await inbox.start(agent)

        assert inbox._server is not None
        bound_path = agent.messaging_socket_path
        assert bound_path == inbox._server.path
        assert callable(agent.provider)
        await inbox.stop()
        assert inbox._server is None
        assert bound_path is not None and not Path(bound_path).exists(), "stop unlinks the socket file"
    finally:
        import shutil

        shutil.rmtree(state_root, ignore_errors=True)


@pytest.mark.asyncio
async def test_disabled_gate_binds_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MESSAGING_ENV_FLAG, "off")
    state_root = _short_state_root()
    try:
        inbox, _store, _session = _inbox(state_root)
        agent = _FakeAgent()

        await inbox.start(agent)

        assert inbox._server is None
        assert agent.messaging_socket_path is None
    finally:
        import shutil

        shutil.rmtree(state_root, ignore_errors=True)


@pytest.mark.asyncio
async def test_start_helper_returns_none_when_not_a_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(MESSAGING_ENV_FLAG, raising=False)
    state_root = _short_state_root()
    try:
        inbox = await start_headless_peer_inbox(
            store_root=state_root,
            session_id="s",
            agent=_FakeAgent(),
            settings=CliSettings(),
            permission_mode_value="ask",
            json_mode=True,
            journal_factory=lambda: None,
            enabled=False,
        )
        assert inbox is None
    finally:
        import shutil

        shutil.rmtree(state_root, ignore_errors=True)


@pytest.mark.asyncio
async def test_accepted_delivery_queues_and_records_events() -> None:
    inbox, store, session = _inbox(_short_state_root())
    agent = _FakeAgent()
    await inbox.start(agent)
    try:
        response = await send_over_socket(
            inbox._server.path,  # type: ignore[attr-defined]
            {
                "v": 1,
                "kind": "message",
                "sender_id": "remote-1",
                "sender_title": "remote",
                "text": "hello worker",
            },
        )
        assert response == {"ok": True, "message_id": response["message_id"], "outcome": "accepted"}

        texts = inbox.pending_texts()
        assert len(texts) == 1
        assert "[Peer message from session 'remote'" in texts[0]
        assert texts[0].endswith("hello worker")

        inputs = await agent.provider()
        assert len(inputs) == 1
        assert inputs[0].origin == {"kind": "peer", "session_id": "remote-1", "title": "remote"}
        assert inbox.pending_texts() == [], "pop consumes the queue exactly once"

        events = await FileSessionEventStore(store.journal(session.session_id)).read(
            session.session_id,
            types={KnownEventType.PEER_MESSAGE_RECEIVED, KnownEventType.PEER_MESSAGE_DELIVERED},
        )
        assert [event.event_type for event in events] == [
            KnownEventType.PEER_MESSAGE_RECEIVED,
            KnownEventType.PEER_MESSAGE_DELIVERED,
        ]
    finally:
        await inbox.stop()


@pytest.mark.asyncio
async def test_hold_policy_drops_explicitly_instead_of_parking() -> None:
    """No interactive surface exists on a worker; a hold can never be answered."""
    inbox, store, session = _inbox(_short_state_root(), settings=CliSettings(cross_session_inbound="hold"))
    await inbox.start(_FakeAgent())
    try:
        response = await send_over_socket(
            inbox._server.path,  # type: ignore[attr-defined]
            {"v": 1, "kind": "message", "sender_id": "r", "sender_title": "remote", "text": "hi"},
        )
        assert response["ok"] is True
        assert response["outcome"] == "refused"
        assert inbox.pending_texts() == []

        received = await FileSessionEventStore(store.journal(session.session_id)).read(
            session.session_id, types={KnownEventType.PEER_MESSAGE_RECEIVED}
        )
        delivered = await FileSessionEventStore(store.journal(session.session_id)).read(
            session.session_id, types={KnownEventType.PEER_MESSAGE_DELIVERED}
        )
        assert len(received) == 1
        assert delivered == []
    finally:
        await inbox.stop()


@pytest.mark.asyncio
async def test_status_query_reflects_busy_flag() -> None:
    inbox, _store, _session = _inbox(_short_state_root())
    await inbox.start(_FakeAgent())
    try:
        path = inbox._server.path  # type: ignore[attr-defined]
        idle = await send_over_socket(path, {"v": 1, "kind": "status"})
        assert idle == {"ok": True, "status": "idle"}

        inbox.mark_busy()
        busy = await send_over_socket(path, {"v": 1, "kind": "status"})
        assert busy == {"ok": True, "status": "busy"}
    finally:
        await inbox.stop()


@pytest.mark.asyncio
async def test_worker_is_reachable_from_another_process() -> None:
    """The bound socket answers status queries addressed by file path alone."""
    import sys

    inbox, _store, _session = _inbox(_short_state_root())
    await inbox.start(_FakeAgent())
    path = str(inbox._server.path)  # type: ignore[attr-defined]
    try:
        program = (
            "import sys;"
            f"sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r});"
            "from pathlib import Path;"
            "from kolega_code.session.inbox import send_over_socket;"
            "import asyncio;"
            "resp = asyncio.run(send_over_socket(Path(sys.argv[1]), {'v': 1, 'kind': 'status'}));"
            "print(resp)"
        )
        # Async spawn: a blocking wait here would freeze this process's own
        # event loop — i.e., the very server the child is trying to reach.
        import asyncio as _asyncio

        proc = await _asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            program,
            path,
            stdout=_asyncio.subprocess.PIPE,
            stderr=_asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await _asyncio.wait_for(proc.communicate(), timeout=30)
        assert proc.returncode == 0, stderr.decode()
        assert "'status': 'idle'" in _stdout.decode()
    finally:
        await inbox.stop()
