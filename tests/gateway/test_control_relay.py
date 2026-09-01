"""ControlRelay: control-channel prompts rendered as buttons and answered by taps."""

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.gateway.adapters.base import AdapterCapabilities, ButtonOption, ChatRef, GatewayAdapter, InboundMessage
from kolega_code.gateway.control_relay import ControlRelay
from kolega_code.gateway.event_router import EventRouter
from kolega_code.permissions import PermissionKind, PermissionMode, PermissionRequest, allow_rule_options
from kolega_code.session.control import ControlChannel
from kolega_code.session.runtime import SessionRuntime, serialize_permission_request


class ButtonAdapter(GatewayAdapter):
    name = "buttony"

    def __init__(self) -> None:
        super().__init__()
        self.capabilities = AdapterCapabilities(supports_edits=True, supports_inline_buttons=True)
        self.buttons: list[tuple[str, str, list[ButtonOption]]] = []
        self.tokens: list[str] = []

    async def send_buttons(self, chat_id: str, prompt: str, options: list[ButtonOption]) -> str:
        await asyncio.sleep(0)
        token = f"tok-{len(self.tokens) + 1}"
        self.tokens.append(token)
        self.buttons.append((chat_id, prompt, list(options)))
        return token


def make_request() -> PermissionRequest:
    return PermissionRequest(
        kind=PermissionKind.COMMAND,
        tool_name="bash",
        inputs={},
        command="ls -la",
        path="",
        mcp_server="",
        mcp_tool="",
    )


def rule_options_payload(request: PermissionRequest) -> list[dict[str, Any]]:
    return [
        {"label": option.label, "description": option.description, "rule": option.rule.to_dict()}
        for option in allow_rule_options(request)
    ]


async def wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


async def make_relay(
    tmp_path: Path,
) -> tuple[ControlRelay, EventRouter, ControlChannel, ButtonAdapter, SessionRuntime]:
    adapter = ButtonAdapter()
    source: asyncio.Queue[AgentEvent] = asyncio.Queue()

    async def emit(event: AgentEvent) -> None:
        await source.put(event)

    channel = ControlChannel(session_id="s-1", emit=emit)
    router = EventRouter(source)
    control_events = router.subscribe(KnownEventType.CONTROL_REQUESTED, KnownEventType.CONTROL_RESOLVED)
    runtime = SessionRuntime(
        session_id="s-1",
        project_path=tmp_path,
        control=channel,
        permission_mode=PermissionMode.ASK,
    )
    relay = ControlRelay(
        chat_ref=ChatRef("buttony", "42"),
        runtime=runtime,
        adapter=adapter,
        event_queue=control_events,
        client_id="buttony:42",
    )
    router.start()
    relay.start()
    return relay, router, channel, adapter, runtime


def tap(token: str, option: str) -> InboundMessage:
    return InboundMessage(
        channel="buttony",
        chat_id="42",
        sender_id="7",
        message_id="tap-1",
        callback_token=token,
        callback_option=option,
    )


@pytest.mark.asyncio
async def test_permission_allow_once(tmp_path: Path) -> None:
    relay, router, channel, adapter, _ = await make_relay(tmp_path)
    try:
        request = make_request()
        task = asyncio.create_task(
            channel.request(
                "permission",
                {"request": serialize_permission_request(request), "rule_options": rule_options_payload(request)},
                default={"allowed": False, "reason": "unanswered"},
            )
        )
        assert await wait_for(lambda: bool(adapter.buttons))
        prompt, options = adapter.buttons[0][1], adapter.buttons[0][2]
        assert "ls -la" in prompt
        assert [option.option_id for option in options][0] == "allow_once"
        assert [option.option_id for option in options][-1] == "deny"
        await relay.handle_tap(tap(adapter.tokens[0], "allow_once"))
        response = await task
        assert response["allowed"] is True
    finally:
        await relay.stop()
        await router.stop()


@pytest.mark.asyncio
async def test_permission_deny(tmp_path: Path) -> None:
    relay, router, channel, adapter, _ = await make_relay(tmp_path)
    try:
        request = make_request()
        task = asyncio.create_task(
            channel.request(
                "permission",
                {"request": serialize_permission_request(request), "rule_options": rule_options_payload(request)},
                default={"allowed": True, "reason": "unanswered"},
            )
        )
        assert await wait_for(lambda: bool(adapter.buttons))
        await relay.handle_tap(tap(adapter.tokens[0], "deny"))
        response = await task
        assert response["allowed"] is False
        assert "Denied" in str(response["reason"])
    finally:
        await relay.stop()
        await router.stop()


@pytest.mark.asyncio
async def test_permission_allow_always_persists_the_rule(tmp_path: Path) -> None:
    relay, router, channel, adapter, _ = await make_relay(tmp_path)
    try:
        request = make_request()
        task = asyncio.create_task(
            channel.request(
                "permission",
                {"request": serialize_permission_request(request), "rule_options": rule_options_payload(request)},
                default={"allowed": False, "reason": "unanswered"},
            )
        )
        assert await wait_for(lambda: bool(adapter.buttons))
        await relay.handle_tap(tap(adapter.tokens[0], "allow_always:0"))
        response = await task
        assert response["allowed"] is True
        assert isinstance(response.get("rule"), dict)
    finally:
        await relay.stop()
        await router.stop()


@pytest.mark.asyncio
async def test_question_prompt_and_answer(tmp_path: Path) -> None:
    relay, router, channel, adapter, _ = await make_relay(tmp_path)
    try:
        task = asyncio.create_task(
            channel.request(
                "question",
                {"question": "Which approach?", "options": ["red", "green"], "descriptions": ["warm", "cool"]},
                default={"answer": None},
            )
        )
        assert await wait_for(lambda: bool(adapter.buttons))
        prompt, options = adapter.buttons[0][1], adapter.buttons[0][2]
        assert prompt == "Which approach?"
        assert [option.label for option in options] == ["red", "green"]
        await relay.handle_tap(tap(adapter.tokens[0], "1"))
        response = await task
        assert response == {"answer": "green"}
    finally:
        await relay.stop()
        await router.stop()


@pytest.mark.asyncio
async def test_stale_or_unknown_taps_are_ignored(tmp_path: Path) -> None:
    relay, router, channel, adapter, _ = await make_relay(tmp_path)
    try:
        request = make_request()
        task = asyncio.create_task(
            channel.request(
                "permission",
                {"request": serialize_permission_request(request), "rule_options": rule_options_payload(request)},
                default={"allowed": False, "reason": "unanswered"},
            )
        )
        assert await wait_for(lambda: bool(adapter.buttons))
        await relay.handle_tap(tap("no-such-token", "allow_once"))
        await relay.handle_tap(tap(adapter.tokens[0], "allow_once"))
        await relay.handle_tap(tap(adapter.tokens[0], "deny"))  # second tap: token already consumed
        response = await task
        assert response["allowed"] is True
    finally:
        await relay.stop()
        await router.stop()


@pytest.mark.asyncio
async def test_releasing_the_lease_resolves_prompts_to_defaults(tmp_path: Path) -> None:
    relay, router, channel, adapter, _ = await make_relay(tmp_path)
    request = make_request()
    task = asyncio.create_task(
        channel.request(
            "permission",
            {"request": serialize_permission_request(request), "rule_options": rule_options_payload(request)},
            default={"allowed": False, "reason": "unanswered"},
        )
    )
    assert await wait_for(lambda: bool(adapter.buttons))
    await relay.stop()
    await router.stop()
    response = await task
    assert response["allowed"] is False
    assert response["reason"] == "unanswered"
