from __future__ import annotations

import asyncio
import io
import json
import os
import time
from pathlib import Path
from typing import Any, cast

import pytest

from kolega_code.browser_extension.framing import read_message
from kolega_code.browser_extension.native_host import (
    NativeHostBroker,
    NativeHostConfig,
    NativeHostConfigurationError,
    NativeMessageEndpoint,
    load_host_config,
)
from kolega_code.browser_extension.multiplex import MultiplexedPeer, RequestTimeoutError
from kolega_code.browser_extension.protocol import Envelope, MessageDirection, ProtocolValidationError
from kolega_code.browser_extension.registry import RuntimeDescriptor, RuntimeDescriptorRegistry
from kolega_code.browser_extension.runtime import RuntimeChannel

ORIGIN = f"chrome-extension://{'a' * 32}/"
OTHER_ORIGIN = f"chrome-extension://{'b' * 32}/"
NOW = int(time.time() * 1000)


def descriptor(runtime_id: str, origin: str = ORIGIN) -> RuntimeDescriptor:
    return RuntimeDescriptor(
        runtime_id=runtime_id,
        session_id=f"session_{runtime_id[-1]}",
        transport="unix",
        endpoint=f"/tmp/{runtime_id}.sock",
        token="x" * 32,
        pid=os.getpid(),
        created_at_ms=NOW,
        expires_at_ms=NOW + 10_000,
        extension_origin=origin,
    )


class FakeEndpoint:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def write(self, envelope: Envelope) -> None:
        self.messages.append(cast(dict[str, Any], envelope.to_mapping()))

    async def write_mapping(self, message: dict[str, object]) -> None:
        self.messages.append(dict(message))


class FakeRegistry:
    def __init__(self, descriptors: list[RuntimeDescriptor]) -> None:
        self.descriptors = {item.runtime_id: item for item in descriptors}

    def list_active(self) -> list[RuntimeDescriptor]:
        return list(self.descriptors.values())

    def read(self, runtime_id: str) -> RuntimeDescriptor:
        return self.descriptors[runtime_id]


class FakeChannel:
    outbound_direction = MessageDirection.RUNTIME_TO_EXTENSION
    runtime_id = "runtime_1"
    session_id = "session_1"

    def __init__(self) -> None:
        self.writes: list[Envelope] = []
        self.closed = False
        self.block = asyncio.Event()

    async def write(self, envelope: Envelope) -> None:
        self.writes.append(envelope)

    async def read(self) -> Envelope | None:
        await self.block.wait()
        return None

    async def close(self) -> None:
        self.closed = True
        self.block.set()


def event_for(item: RuntimeDescriptor) -> Envelope:
    return Envelope.event(
        direction=MessageDirection.EXTENSION_TO_RUNTIME,
        request_id=f"event_{item.runtime_id}",
        runtime_id=item.runtime_id,
        session_id=item.session_id,
        deadline_ms=NOW + 10_000,
        event="browser.session_ready",
        data={},
    )


def test_host_config_is_exact_private_and_origin_bound(tmp_path: Path) -> None:
    path = tmp_path / "host.json"
    value = {
        "extension_origin": ORIGIN,
        "registry_dir": str(tmp_path / "runtimes"),
        "max_runtimes": 4,
        "max_pending_requests": 8,
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)
    assert load_host_config(path) == NativeHostConfig(
        extension_origin=ORIGIN,
        registry_dir=tmp_path / "runtimes",
        max_runtimes=4,
        max_pending_requests=8,
    )
    path.write_text(json.dumps({**value, "extra": True}), encoding="utf-8")
    with pytest.raises(NativeHostConfigurationError, match="invalid schema"):
        load_host_config(path)
    path.write_text(json.dumps(value), encoding="utf-8")
    if hasattr(os, "getuid"):
        path.chmod(0o644)
        with pytest.raises(NativeHostConfigurationError, match="permissions"):
            load_host_config(path)


@pytest.mark.asyncio
async def test_native_endpoint_serializes_concurrent_complete_frames() -> None:
    output = io.BytesIO()
    endpoint = NativeMessageEndpoint(io.BytesIO(), output, max_pending_writes=2)
    await asyncio.gather(
        endpoint.write_mapping({"sequence": 1}),
        endpoint.write_mapping({"sequence": 2}),
    )
    await endpoint.close()
    stream = io.BytesIO(output.getvalue())
    first = read_message(stream)
    second = read_message(stream)
    assert first is not None
    assert second is not None
    assert {first["sequence"], second["sequence"]} == {1, 2}
    assert read_message(stream) is None


@pytest.mark.asyncio
async def test_broker_discovers_only_exact_origin_and_routes_multiple_runtimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = descriptor("runtime_1")
    second = descriptor("runtime_2")
    foreign = descriptor("runtime_3", OTHER_ORIGIN)
    endpoint = FakeEndpoint()
    registry = FakeRegistry([first, second, foreign])
    channels: dict[str, FakeChannel] = {}

    async def connect(item: RuntimeDescriptor, *, extension_origin: str, timeout: float) -> RuntimeChannel:
        assert extension_origin == ORIGIN
        assert timeout > 0
        channel = channels.setdefault(item.runtime_id, FakeChannel())
        return cast(RuntimeChannel, channel)

    monkeypatch.setattr("kolega_code.browser_extension.native_host.connect_runtime_channel", connect)
    broker = NativeHostBroker(
        endpoint=cast(NativeMessageEndpoint, endpoint),
        registry=cast(RuntimeDescriptorRegistry, registry),
        extension_origin=ORIGIN,
    )
    await broker._list_runtimes({"kind": "list_runtimes", "protocol_version": 1, "request_id": "discover_1"})
    assert endpoint.messages[0]["kind"] == "runtimes"
    assert [item["runtime_id"] for item in endpoint.messages[0]["runtimes"]] == ["runtime_1", "runtime_2"]

    await broker._route_from_extension(event_for(first))
    await broker._route_from_extension(event_for(second))
    assert channels["runtime_1"].writes == [event_for(first)]
    assert channels["runtime_2"].writes == [event_for(second)]
    await broker.close()
    await broker.close()
    assert all(channel.closed for channel in channels.values())


@pytest.mark.asyncio
async def test_broker_correlates_both_directions_and_enforces_deadlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = descriptor("runtime_1")
    endpoint = FakeEndpoint()
    channel = FakeChannel()

    async def connect(runtime: RuntimeDescriptor, *, extension_origin: str, timeout: float) -> RuntimeChannel:
        assert runtime == item
        assert extension_origin == ORIGIN
        assert timeout > 0
        return cast(RuntimeChannel, channel)

    monkeypatch.setattr("kolega_code.browser_extension.native_host.connect_runtime_channel", connect)
    broker = NativeHostBroker(
        endpoint=cast(NativeMessageEndpoint, endpoint),
        registry=cast(RuntimeDescriptorRegistry, FakeRegistry([item])),
        extension_origin=ORIGIN,
    )
    await broker._route_from_extension(event_for(item))
    relay = broker._runtimes[item.runtime_id]

    far_future_extension_request = Envelope.request(
        direction=MessageDirection.EXTENSION_TO_RUNTIME,
        request_id="far_future_extension_request",
        runtime_id=item.runtime_id,
        session_id=item.session_id,
        deadline_ms=int(time.time() * 1000) + 301_000,
        operation="browser.snapshot",
        params={"target": None, "depth": None},
    )
    writes_before = len(channel.writes)
    await broker._route_from_extension(far_future_extension_request)
    assert endpoint.messages[-1]["payload"]["code"] == "invalid_deadline"
    assert len(channel.writes) == writes_before
    assert broker.pending_count == 0

    far_future_runtime_request = Envelope.request(
        direction=MessageDirection.RUNTIME_TO_EXTENSION,
        request_id="far_future_runtime_request",
        runtime_id=item.runtime_id,
        session_id=item.session_id,
        deadline_ms=int(time.time() * 1000) + 301_000,
        operation="browser.snapshot",
        params={"target": None, "depth": None},
    )
    endpoint_writes_before = len(endpoint.messages)
    await broker._route_from_runtime(relay, far_future_runtime_request)
    assert channel.writes[-1].payload["code"] == "invalid_deadline"
    assert len(endpoint.messages) == endpoint_writes_before
    assert broker.pending_count == 0

    runtime_request = Envelope.request(
        direction=MessageDirection.RUNTIME_TO_EXTENSION,
        request_id="runtime_request",
        runtime_id=item.runtime_id,
        session_id=item.session_id,
        deadline_ms=int(time.time() * 1000) + 1_000,
        operation="browser.snapshot",
        params={"target": None, "depth": None},
    )
    await broker._route_from_runtime(relay, runtime_request)
    assert endpoint.messages[-1] == runtime_request.to_mapping()
    response = Envelope.response_for(runtime_request, {"url": "https://example.com", "snapshot": "page"})
    await broker._route_from_extension(response)
    assert channel.writes[-1] == response
    assert broker.pending_count == 0

    expiring_request = Envelope.request(
        direction=MessageDirection.EXTENSION_TO_RUNTIME,
        request_id="expiring_request",
        runtime_id=item.runtime_id,
        session_id=item.session_id,
        deadline_ms=int(time.time() * 1000) + 10,
        operation="browser.snapshot",
        params={"target": None, "depth": None},
    )
    await broker._route_from_extension(expiring_request)
    assert channel.writes[-1] == expiring_request
    await asyncio.sleep(0.03)
    assert endpoint.messages[-1]["payload"]["code"] == "deadline_exceeded"
    assert channel.writes[-1].type.value == "cancel"
    assert broker.pending_count == 0
    await broker.close()


@pytest.mark.asyncio
async def test_multiplex_rejects_a_response_delivered_after_its_absolute_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = FakeChannel()
    peer = MultiplexedPeer(cast(RuntimeChannel, channel))
    pending = asyncio.create_task(
        peer.request(
            "browser.snapshot",
            {"target": None, "depth": None},
            timeout=1,
            request_id="late_response",
        )
    )
    while not channel.writes:
        await asyncio.sleep(0)
    sent = channel.writes[-1]
    monkeypatch.setattr(
        "kolega_code.browser_extension.multiplex.time.time",
        lambda: (sent.deadline_ms + 1) / 1_000,
    )
    peer._handle_response(Envelope.response_for(sent, {"snapshot": "too late"}))
    with pytest.raises(RequestTimeoutError, match="deadline"):
        await pending
    await peer.close()


@pytest.mark.asyncio
async def test_multiplex_request_deadline_includes_a_blocked_write() -> None:
    class BlockingWriteChannel(FakeChannel):
        async def write(self, envelope: Envelope) -> None:
            self.writes.append(envelope)
            await asyncio.Event().wait()

    channel = BlockingWriteChannel()
    peer = MultiplexedPeer(cast(RuntimeChannel, channel))
    started = time.monotonic()
    with pytest.raises(RequestTimeoutError, match="deadline"):
        await peer.request(
            "browser.snapshot",
            {"target": None, "depth": None},
            timeout=0.02,
            request_id="blocked_write",
        )
    assert time.monotonic() - started < 0.15
    assert peer.closed


@pytest.mark.asyncio
async def test_broker_rejects_a_response_after_the_absolute_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = descriptor("runtime_1")
    endpoint = FakeEndpoint()
    channel = FakeChannel()

    async def connect(runtime: RuntimeDescriptor, *, extension_origin: str, timeout: float) -> RuntimeChannel:
        assert runtime == item
        assert extension_origin == ORIGIN
        assert timeout > 0
        return cast(RuntimeChannel, channel)

    monkeypatch.setattr("kolega_code.browser_extension.native_host.connect_runtime_channel", connect)
    broker = NativeHostBroker(
        endpoint=cast(NativeMessageEndpoint, endpoint),
        registry=cast(RuntimeDescriptorRegistry, FakeRegistry([item])),
        extension_origin=ORIGIN,
    )
    await broker._route_from_extension(event_for(item))
    relay = broker._runtimes[item.runtime_id]
    pending_request = Envelope.request(
        direction=MessageDirection.RUNTIME_TO_EXTENSION,
        request_id="late_broker_response",
        runtime_id=item.runtime_id,
        session_id=item.session_id,
        deadline_ms=int(time.time() * 1000) + 1_000,
        operation="browser.snapshot",
        params={"target": None, "depth": None},
    )
    await broker._route_from_runtime(relay, pending_request)
    object.__setattr__(pending_request, "deadline_ms", int(time.time() * 1000) - 1)
    late_response = Envelope.response_for(pending_request, {"snapshot": "late"})
    await broker._route_from_extension(late_response)
    assert channel.writes[-1].payload["code"] == "deadline_exceeded"
    await broker.close()


def test_broker_revalidates_fixed_operation_requests() -> None:
    envelope = Envelope.request(
        direction=MessageDirection.RUNTIME_TO_EXTENSION,
        request_id="request_1",
        runtime_id="runtime_1",
        session_id="session_1",
        deadline_ms=NOW + 10_000,
        operation="browser.navigate",
        params={"url": "https://example.com"},
    )
    object.__setattr__(envelope, "payload", {"operation": "browser.evaluate", "params": {}})
    with pytest.raises(ProtocolValidationError) as error:
        NativeHostBroker._validate_request(envelope)
    assert error.value.code == "unsupported_operation"
