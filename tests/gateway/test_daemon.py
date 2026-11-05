"""GatewayDaemon: lifecycle, lock, dedup, allowlist, and error isolation."""

import asyncio
import io
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from kolega_code.gateway.access import ALLOWLIST_FILE_NAME, PAIRING_FILE_NAME, GatewayAccessControl
from kolega_code.gateway.adapters.base import ChatRef, GatewayAdapter, InboundMessage
from kolega_code.gateway.adapters.echo import EchoAdapter
from kolega_code.gateway.config import GatewayConfig
from kolega_code.gateway.daemon import ERROR_REPLY, GatewayDaemon, GatewayDaemonError, GatewayShutdownIncomplete


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
            allowed_users=("123",),
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
async def test_stop_retains_cancellation_resistant_dispatch_and_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    order: list[str] = []

    class Handler(RecordingHandler):
        async def handle(self, chat_ref: ChatRef, message: InboundMessage) -> None:
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
            finally:
                order.append("used-after-close" if self.shutdown_calls else "finalized")

        async def shutdown(self) -> None:
            order.append("closed")
            await super().shutdown()

    monkeypatch.setattr("kolega_code.gateway.daemon.SHUTDOWN_WAIT_SECONDS", 0.05)
    adapter, handler = RecordingAdapter(), Handler()
    daemon = GatewayDaemon(make_config(tmp_path), adapter, handler)
    await daemon.start()
    try:
        await adapter.inbound.put(inbound("blocked"))
        await asyncio.wait_for(entered.wait(), 10)
        for _ in range(2):
            with pytest.raises(GatewayShutdownIncomplete):
                await daemon.stop()
        await asyncio.wait_for(cancelled.wait(), 10)
        owner = daemon._stop_task
        assert owner is not None and not owner.done()
        assert not adapter.stopped
        assert daemon._lock is not None and daemon._lock.is_locked
        assert not daemon.is_message_allowed(inbound("new"))
        assert handler.shutdown_calls == 0 and order == []
        assert daemon.status().shutdown_state == "incomplete"
        assert daemon._dispatch_task is not None and daemon._dispatch_task.cancelling() == 1
        lock = daemon._lock
        with pytest.raises(GatewayDaemonError, match="cannot be restarted"):
            await daemon.start()
        assert daemon._lock is lock
        waiter = asyncio.create_task(daemon.stop())
        await asyncio.sleep(0)
        waiter.cancel("stop waiter cancelled")
        with pytest.raises(asyncio.CancelledError, match="stop waiter cancelled"):
            await waiter
        assert daemon._stop_task is owner and not owner.cancelling()
        release.set()
        await asyncio.wait_for(daemon.stop(), 10)
        assert adapter.stopped and handler.shutdown_calls == 1
        assert order == ["finalized", "closed"]
        assert daemon._lock is None and daemon.status().shutdown_state == "stopped"
    finally:
        release.set()
        await daemon.stop()


@pytest.mark.asyncio
async def test_split_shutdown_fences_admission_before_dispatch_drains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, closing, finalized, release = (asyncio.Event() for _ in range(4))
    order: list[str] = []

    class Handler(RecordingHandler):
        def begin_shutdown(self) -> None:
            order.append("admission-closed")

        async def handle(self, chat_ref: ChatRef, message: InboundMessage) -> None:
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                order.append("dispatch-cancelled")
                await closing.wait()
                await release.wait()
            finally:
                order.append("finalized")
                finalized.set()

        async def wait_closed(self) -> None:
            closing.set()
            await finalized.wait()
            order.append("closed")

    monkeypatch.setattr("kolega_code.gateway.daemon.SHUTDOWN_WAIT_SECONDS", 0.05)
    adapter, handler = RecordingAdapter(), Handler()
    daemon = GatewayDaemon(make_config(tmp_path), adapter, handler)
    await daemon.start()
    try:
        await adapter.inbound.put(inbound("blocked"))
        await asyncio.wait_for(entered.wait(), 10)
        with pytest.raises(GatewayShutdownIncomplete):
            await daemon.stop()
        await asyncio.wait_for(closing.wait(), 10)
        assert order == ["admission-closed", "dispatch-cancelled"]
        assert not daemon.is_message_allowed(inbound("new"))
        assert not adapter.stopped and handler.shutdown_calls == 0
        release.set()
        await asyncio.wait_for(daemon.stop(), 10)
        assert order == ["admission-closed", "dispatch-cancelled", "finalized", "closed"]
        assert adapter.stopped and handler.shutdown_calls == 0
    finally:
        closing.set()
        release.set()
        await daemon.stop()


@pytest.mark.asyncio
async def test_stopped_daemon_rejects_restart_before_acquiring_resources(tmp_path: Path) -> None:
    adapter, handler = RecordingAdapter(), RecordingHandler()
    daemon = GatewayDaemon(make_config(tmp_path), adapter, handler)
    await daemon.start()
    await daemon.stop()
    owner = daemon._stop_task
    adapter.started = False
    try:
        with pytest.raises(GatewayDaemonError, match="cannot be restarted"):
            await daemon.start()
        assert not adapter.started
        assert daemon._stop_task is owner
        assert daemon._lock is None and daemon._dispatch_task is None and daemon._heartbeat_task is None
        assert not daemon._status_path.exists() and not daemon._pid_path.exists()
        await daemon.stop()
        assert handler.shutdown_calls == 1
    finally:
        # Keep the regression probe from leaking tasks if restart is incorrectly accepted.
        for task in (daemon._dispatch_task, daemon._heartbeat_task):
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        if daemon._lock is not None:
            await adapter.stop()
            daemon._remove_status()
            daemon._remove_pid()
            daemon._release_lock()


@pytest.mark.asyncio
async def test_running_daemon_rejects_duplicate_start_without_replacing_owners(tmp_path: Path) -> None:
    daemon = GatewayDaemon(make_config(tmp_path), RecordingAdapter(), RecordingHandler())
    await daemon.start()
    try:
        lock, dispatch, heartbeat = daemon._lock, daemon._dispatch_task, daemon._heartbeat_task
        with pytest.raises(GatewayDaemonError, match="already started"):
            await daemon.start()
        assert daemon._lock is lock and daemon._dispatch_task is dispatch and daemon._heartbeat_task is heartbeat
        assert lock is not None and lock.is_locked
        assert daemon.status().running
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_stop_before_start_leaves_daemon_startable(tmp_path: Path) -> None:
    adapter, handler = RecordingAdapter(), RecordingHandler()
    daemon = GatewayDaemon(make_config(tmp_path), adapter, handler)
    await daemon.stop()
    await daemon.start()
    try:
        assert daemon.status().running
        await adapter.inbound.put(inbound("hello"))
        assert await wait_for(lambda: len(handler.handled) == 1, timeout=10)
    finally:
        await daemon.stop()
    assert adapter.stopped and handler.shutdown_calls == 1
    assert daemon._lock is None


@pytest.mark.asyncio
async def test_cancelled_chat_operation_does_not_cancel_dispatch(tmp_path: Path) -> None:
    class Handler(RecordingHandler):
        async def handle(self, chat_ref: ChatRef, message: InboundMessage) -> None:
            if message.text == "cancel":
                raise asyncio.CancelledError("chat operation stopped")
            await super().handle(chat_ref, message)

    adapter, handler = RecordingAdapter(), Handler()
    daemon = GatewayDaemon(make_config(tmp_path), adapter, handler)
    await daemon.start()
    try:
        await adapter.inbound.put(inbound("cancel", "one"))
        await adapter.inbound.put(inbound("other", "two", chat_id="other"))
        assert await wait_for(lambda: len(handler.handled) == 1, timeout=10)
        assert handler.handled[0][1].text == "other"
        assert daemon._dispatch_task is not None and not daemon._dispatch_task.done()
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_lock_loser_stop_does_not_touch_the_owning_daemon_state(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    winner = GatewayDaemon(config, RecordingAdapter(), RecordingHandler())
    loser_adapter, loser_handler = RecordingAdapter(), RecordingHandler()
    loser = GatewayDaemon(config, loser_adapter, loser_handler)
    await winner.start()
    try:
        status, pid = winner._status_path.read_bytes(), winner._pid_path.read_bytes()
        with pytest.raises(GatewayDaemonError, match="Another gateway instance"):
            await loser.start()
        await loser.stop()
        assert winner._status_path.read_bytes() == status
        assert winner._pid_path.read_bytes() == pid
        assert winner._lock is not None and winner._lock.is_locked
        assert not loser_adapter.stopped and loser_handler.shutdown_calls == 0
    finally:
        await winner.stop()
    await loser.start()
    try:
        assert loser.status().running
        assert loser.is_message_allowed(inbound("retry"))
        await loser_adapter.inbound.put(inbound("retry"))
        assert await wait_for(lambda: len(loser_handler.handled) == 1, timeout=10)
    finally:
        await loser.stop()
    assert loser_adapter.stopped and loser_handler.shutdown_calls == 1
    assert loser._lock is None
    assert not loser._status_path.exists() and not loser._pid_path.exists()


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_recreate_status_after_successful_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon = GatewayDaemon(make_config(tmp_path), RecordingAdapter(), RecordingHandler())
    await daemon.start()
    await daemon.stop()
    task = asyncio.current_task()
    assert task is not None

    async def cancelled_wait(*args: Any, **kwargs: Any) -> Any:
        raise asyncio.CancelledError("waiter interrupted after cleanup")

    monkeypatch.setattr("kolega_code.gateway.daemon.asyncio.wait", cancelled_wait)
    with pytest.raises(asyncio.CancelledError, match="waiter interrupted after cleanup"):
        await daemon.stop()
    assert daemon.status().shutdown_state == "stopped"
    assert not daemon._status_path.exists() and not daemon._pid_path.exists()


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
@pytest.mark.parametrize("allowed_users", [(), ("123",)])
async def test_pairing_replies_with_a_code_and_admission_works(tmp_path: Path, allowed_users: tuple[str, ...]) -> None:
    adapter = RecordingAdapter()
    handler = RecordingHandler()
    config = make_config(tmp_path, allowed_users=allowed_users, pairing_enabled=True)
    daemon = GatewayDaemon(config, adapter, turn_handler=handler)
    await daemon.start()
    try:
        await adapter.inbound.put(inbound("hello?", message_id="m-1", sender_id="999"))
        assert await wait_for(lambda: len(adapter.sent) == 1)
        reply = adapter.sent[0][1]
        assert "pairing approve" in reply
        code = reply.rsplit(" ", 1)[-1]
        assert handler.handled == []
        assert not daemon.is_message_allowed(inbound("before approval", sender_id="999"))

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
        assert not daemon.is_message_allowed(inbound("other outsider", sender_id="555"))
        assert daemon.status().access["paired_users"] == 1
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
@pytest.mark.parametrize("pairing_enabled", [False, True])
async def test_empty_allowlist_denies_everyone(tmp_path: Path, pairing_enabled: bool) -> None:
    adapter = RecordingAdapter()
    handler = RecordingHandler()
    config = make_config(tmp_path, allowed_users=(), pairing_enabled=pairing_enabled)
    daemon = GatewayDaemon(config, adapter, turn_handler=handler)
    await daemon.start()
    try:
        # Await actual dispatch completion rather than timing out on an absence.
        await daemon._handle_message(inbound("anyone", sender_id="999"))
        assert handler.handled == []
        assert len(adapter.sent) == int(pairing_enabled)
        assert daemon.status().access["policy"] == ("pairing-only" if pairing_enabled else "locked")
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
        assert payload["access"] == {
            "policy": "restricted",
            "configured_users": 1,
            "paired_users": 0,
            "pairing_enabled": False,
        }
    finally:
        await daemon.stop()
    assert not (tmp_path / "state" / "gateway.status.json").exists()


@pytest.mark.asyncio
async def test_rejected_callbacks_never_issue_pairing(tmp_path: Path) -> None:
    adapter, handler = RecordingAdapter(), RecordingHandler()
    daemon = GatewayDaemon(make_config(tmp_path, allowed_users=(), pairing_enabled=True), adapter, handler)
    await daemon._handle_message(
        replace(inbound("", sender_id="999"), callback_token="pending-token", callback_option="allow_once")
    )
    assert handler.handled == []
    assert adapter.sent == []
    assert not (tmp_path / "state" / PAIRING_FILE_NAME).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", [PAIRING_FILE_NAME, ALLOWLIST_FILE_NAME])
@pytest.mark.parametrize(
    "content",
    ['{"bad": null, "other": {"expires_at": []}}', '{"999":' + "[" * 2000 + "0" + "]" * 2000 + "}"],
)
async def test_corrupt_access_data_does_not_stop_dispatch(tmp_path: Path, filename: str, content: str) -> None:
    config = make_config(tmp_path, pairing_enabled=True)
    config.state_dir.mkdir()
    (config.state_dir / filename).write_text(content)
    adapter, handler = RecordingAdapter(), RecordingHandler()
    daemon = GatewayDaemon(config, adapter, handler)
    await daemon.start()
    try:
        await adapter.inbound.put(inbound("pair me", sender_id="999"))
        await adapter.inbound.put(inbound("trusted", message_id="m-2"))
        assert await wait_for(lambda: len(handler.handled) == 1)
        assert len(adapter.sent) == 1
        assert handler.handled[0][1].sender_id == "123"
        assert daemon.status().access["configured_users"] == 1
        daemon._write_status()
    finally:
        await daemon.stop()


def test_adapter_preflight_uses_the_same_sender_and_group_policy(tmp_path: Path) -> None:
    adapter = RecordingAdapter()
    daemon = GatewayDaemon(make_config(tmp_path, group_ids=("-100",)), adapter, RecordingHandler())
    assert adapter.is_inbound_allowed(inbound("owner"))
    assert not adapter.is_inbound_allowed(inbound("unknown", sender_id="999"))
    assert not adapter.is_inbound_allowed(inbound("ambient", is_group=True, chat_id="-100"))
    assert not adapter.is_inbound_allowed(inbound("unlisted", is_group=True, bot_mentioned=True, chat_id="-999"))
    assert adapter.is_inbound_allowed(inbound("listed", is_group=True, bot_mentioned=True, chat_id="-100"))
    assert not daemon.is_message_allowed(
        inbound("listed but unknown", sender_id="999", is_group=True, bot_mentioned=True, chat_id="-100")
    )


@pytest.mark.parametrize("allowed_users", [(), ("custom",)])
def test_echo_allowance_is_scoped_to_actual_console_transport(tmp_path: Path, allowed_users: tuple[str, ...]) -> None:
    config = make_config(tmp_path, adapter="echo", allowed_users=allowed_users)
    echo = EchoAdapter(stdin=io.StringIO(), stdout=io.StringIO())
    daemon = GatewayDaemon(config, echo, RecordingHandler())
    assert daemon.is_message_allowed(inbound("console", sender_id="owner")) == (not allowed_users)
    assert daemon.is_message_allowed(inbound("custom", sender_id="custom")) == bool(allowed_users)
    assert not daemon.is_message_allowed(inbound("stranger", sender_id="999"))
    assert daemon.status().access["policy"] == ("restricted" if allowed_users else "local-echo")

    # Neither a spoofed message channel nor a configured adapter name is enough.
    remote = GatewayDaemon(config, RecordingAdapter(), RecordingHandler())
    assert not remote.is_message_allowed(replace(inbound("spoof", sender_id="owner"), channel="echo"))


@pytest.mark.asyncio
async def test_locked_start_warns_once_and_stays_running(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    daemon = GatewayDaemon(make_config(tmp_path, allowed_users=()), RecordingAdapter(), RecordingHandler())
    await daemon.start()
    try:
        assert daemon.status().running
        assert caplog.text.count("access locked") == 1
        daemon._write_status()
        assert caplog.text.count("access locked") == 1
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_status_uses_startup_configuration_and_live_pairing_file(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    daemon = GatewayDaemon(config, RecordingAdapter(), RecordingHandler())
    await daemon.start()
    try:
        # Editing disk settings does not change this process's configured ACL.
        (config.state_dir / "settings.json").write_text('{"gateway": {"allowed_users": []}}')
        (config.state_dir / ALLOWLIST_FILE_NAME).write_text(
            json.dumps({"999": {"name": "Paired", "approved_at": 1000}})
        )
        daemon._write_status()
        payload = json.loads((config.state_dir / "gateway.status.json").read_text())
        assert payload["access"] == {
            "policy": "restricted",
            "configured_users": 1,
            "paired_users": 1,
            "pairing_enabled": False,
        }
        assert daemon.is_message_allowed(inbound("configured"))
        assert daemon.is_message_allowed(inbound("paired", sender_id="999"))
    finally:
        await daemon.stop()
