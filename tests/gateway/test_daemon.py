"""GatewayDaemon: lifecycle, lock, dedup, allowlist, and error isolation."""

import asyncio
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from kolega_code.gateway.access import GatewayAccessControl
from kolega_code.gateway.adapters.base import ChatRef, GatewayAdapter, InboundMessage
from kolega_code.gateway.config import GatewayConfig
from kolega_code.gateway.daemon import ERROR_REPLY, GatewayDaemon, GatewayDaemonError


class RecordingAdapter(GatewayAdapter):
    name = "recording"

    def __init__(self) -> None:
        super().__init__()
        self.sent: list[tuple[str, str]] = []
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def health(self) -> dict[str, Any]:
        return {"state": "running" if self.started else "stopped"}

    async def send_text(self, chat_id: str, text: str, *, reply_to_message_id: str | None = None) -> str:
        self.sent.append((chat_id, text))
        return f"out-{len(self.sent)}"


class RecordingHandler:
    def __init__(self, *, fail_for: str | None = None) -> None:
        self.handled: list[tuple[ChatRef, InboundMessage]] = []
        self.shutdown_calls = 0
        self._fail_for = fail_for

    def status(self) -> dict[str, Any]:
        return {"active_sessions": len(self.handled)}

    async def handle(self, chat_ref: ChatRef, message: InboundMessage) -> None:
        if self._fail_for is not None and message.text == self._fail_for:
            raise RuntimeError("boom")
        self.handled.append((chat_ref, message))

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


def make_config(tmp_path: Path, **overrides: Any) -> GatewayConfig:
    return replace(
        GatewayConfig(
            adapter="recording",
            project_path=tmp_path / "workspace",
            state_dir=tmp_path / "state",
        ),
        **overrides,
    )


def inbound(
    text: str,
    message_id: str = "m-1",
    sender_id: str = "123",
    *,
    is_group: bool = False,
    bot_mentioned: bool = False,
    chat_id: str = "chat-1",
) -> InboundMessage:
    return InboundMessage(
        channel="recording",
        chat_id=chat_id,
        sender_id=sender_id,
        message_id=message_id,
        text=text,
        is_group=is_group,
        bot_mentioned=bot_mentioned,
    )


async def wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


@pytest.mark.asyncio
async def test_dispatch_runs_the_turn_handler(tmp_path: Path) -> None:
    adapter = RecordingAdapter()
    handler = RecordingHandler()
    daemon = GatewayDaemon(make_config(tmp_path), adapter, turn_handler=handler)
    await daemon.start()
    try:
        await adapter.inbound.put(inbound("hello", message_id="m-1"))
        assert await wait_for(lambda: len(handler.handled) == 1)
        chat_ref, message = handler.handled[0]
        assert chat_ref.key == "recording:chat-1"
        assert message.text == "hello"
    finally:
        await daemon.stop()
    assert adapter.stopped is True
    assert handler.shutdown_calls == 1


@pytest.mark.asyncio
async def test_duplicate_message_ids_are_dropped(tmp_path: Path) -> None:
    adapter = RecordingAdapter()
    handler = RecordingHandler()
    daemon = GatewayDaemon(make_config(tmp_path), adapter, turn_handler=handler)
    await daemon.start()
    try:
        await adapter.inbound.put(inbound("first", message_id="m-1"))
        await adapter.inbound.put(inbound("retry", message_id="m-1"))
        assert await wait_for(lambda: len(handler.handled) == 1)
        await asyncio.sleep(0.05)
        assert len(handler.handled) == 1
        assert handler.handled[0][1].text == "first"
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_allowlist_blocks_unknown_senders_silently(tmp_path: Path) -> None:
    adapter = RecordingAdapter()
    handler = RecordingHandler()
    daemon = GatewayDaemon(make_config(tmp_path, allowed_users=("123",)), adapter, turn_handler=handler)
    await daemon.start()
    try:
        await adapter.inbound.put(inbound("intruder", message_id="m-1", sender_id="999"))
        await adapter.inbound.put(inbound("owner", message_id="m-2", sender_id="123"))
        assert await wait_for(lambda: len(handler.handled) == 1)
        assert handler.handled[0][1].sender_id == "123"
        # Unauthorized senders get nothing — no reply, no turn.
        assert adapter.sent == []
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_pairing_replies_with_a_code_and_admission_works(tmp_path: Path) -> None:
    adapter = RecordingAdapter()
    handler = RecordingHandler()
    config = make_config(tmp_path, allowed_users=("123",), pairing_enabled=True)
    daemon = GatewayDaemon(config, adapter, turn_handler=handler)
    await daemon.start()
    try:
        await adapter.inbound.put(inbound("hello?", message_id="m-1", sender_id="999"))
        assert await wait_for(lambda: len(adapter.sent) == 1)
        reply = adapter.sent[0][1]
        assert "pairing approve" in reply
        code = reply.rsplit(" ", 1)[-1]

        # The operator approves the code from another process (the CLI path).
        access = GatewayAccessControl(
            state_dir=config.state_dir,
            allowed_users=config.allowed_users,
            pairing_enabled=True,
        )
        assert access.approve(code) == "999"

        # The daemon re-reads on the next message; the sender is now admitted.
        await adapter.inbound.put(inbound("hello again", message_id="m-2", sender_id="999"))
        assert await wait_for(lambda: len(handler.handled) == 1)
        assert handler.handled[0][1].sender_id == "999"
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_group_policy_requires_mention_and_listed_group(tmp_path: Path) -> None:
    adapter = RecordingAdapter()
    handler = RecordingHandler()
    config = make_config(tmp_path, allowed_users=("123",), group_ids=("-100",))
    daemon = GatewayDaemon(config, adapter, turn_handler=handler)
    await daemon.start()
    try:
        # Unaddressed group chatter is dropped.
        await adapter.inbound.put(inbound("ambient noise", message_id="m-1", is_group=True, chat_id="-100"))
        # An unlisted group is dropped even when mentioned.
        await adapter.inbound.put(
            inbound("@bot hello", message_id="m-2", is_group=True, bot_mentioned=True, chat_id="-999")
        )
        # A mention in a listed group reaches the turn handler.
        await adapter.inbound.put(
            inbound("@bot do the thing", message_id="m-3", is_group=True, bot_mentioned=True, chat_id="-100")
        )
        assert await wait_for(lambda: len(handler.handled) == 1)
        assert handler.handled[0][1].text == "@bot do the thing"
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_all_groups_allowed_when_no_group_list_configured(tmp_path: Path) -> None:
    adapter = RecordingAdapter()
    handler = RecordingHandler()
    daemon = GatewayDaemon(make_config(tmp_path, allowed_users=("123",)), adapter, turn_handler=handler)
    await daemon.start()
    try:
        await adapter.inbound.put(inbound("hi", message_id="m-1", is_group=True, bot_mentioned=True, chat_id="-42"))
        assert await wait_for(lambda: len(handler.handled) == 1)
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_empty_allowlist_allows_everyone(tmp_path: Path) -> None:
    adapter = RecordingAdapter()
    handler = RecordingHandler()
    daemon = GatewayDaemon(make_config(tmp_path), adapter, turn_handler=handler)
    await daemon.start()
    try:
        await adapter.inbound.put(inbound("anyone", message_id="m-1", sender_id="999"))
        assert await wait_for(lambda: len(handler.handled) == 1)
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_handler_failure_sends_error_reply_and_keeps_serving(tmp_path: Path) -> None:
    adapter = RecordingAdapter()
    handler = RecordingHandler(fail_for="explode")
    daemon = GatewayDaemon(make_config(tmp_path), adapter, turn_handler=handler)
    await daemon.start()
    try:
        await adapter.inbound.put(inbound("explode", message_id="m-1"))
        assert await wait_for(lambda: len(adapter.sent) == 1)
        assert adapter.sent[0][1] == ERROR_REPLY
        # The dispatch loop survives and processes the next message.
        await adapter.inbound.put(inbound("fine", message_id="m-2"))
        assert await wait_for(lambda: len(handler.handled) == 1)
        assert handler.handled[0][1].text == "fine"
        assert daemon.status().recent_errors == 1
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_second_daemon_on_same_state_dir_is_refused(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    daemon_one = GatewayDaemon(config, RecordingAdapter(), turn_handler=RecordingHandler())
    daemon_two = GatewayDaemon(config, RecordingAdapter(), turn_handler=RecordingHandler())
    await daemon_one.start()
    try:
        with pytest.raises(GatewayDaemonError):
            await daemon_two.start()
    finally:
        await daemon_one.stop()
    # After a clean stop the lock is free again.
    await daemon_two.start()
    await daemon_two.stop()


@pytest.mark.asyncio
async def test_start_creates_workspace_and_state_dirs(tmp_path: Path) -> None:
    workspace = tmp_path / "nested" / "workspace"
    daemon = GatewayDaemon(make_config(tmp_path, project_path=workspace), RecordingAdapter(), RecordingHandler())
    await daemon.start()
    try:
        assert workspace.is_dir()
        assert (tmp_path / "state" / "gateway.lock").exists()
        assert (tmp_path / "state" / "gateway.pid").exists()
        status = daemon.status()
        assert status.running is True
        assert status.adapter == "recording"
        assert status.pid is not None
    finally:
        await daemon.stop()
    assert not (tmp_path / "state" / "gateway.pid").exists()


@pytest.mark.asyncio
async def test_heartbeat_writes_and_removes_the_status_file(tmp_path: Path) -> None:
    import json

    daemon = GatewayDaemon(make_config(tmp_path), RecordingAdapter(), RecordingHandler())
    await daemon.start()
    try:
        status_path = tmp_path / "state" / "gateway.status.json"
        assert status_path.exists()
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        assert payload["running"] is True
        assert payload["adapter"] == "recording"
        assert payload["active_sessions"] == 0
        assert isinstance(payload["heartbeat_at"], str)
        assert payload["pid"] == daemon.status().pid
    finally:
        await daemon.stop()
    assert not (tmp_path / "state" / "gateway.status.json").exists()
