import asyncio
from collections.abc import AsyncIterator
from contextlib import nullcontext
from functools import partial
import inspect
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
from openai import APIError, APIStatusError, AsyncOpenAI
import pytest

from kolega_code.llm.instrumented_client import InstrumentedLLMClient, MinimalLangfuseStreamWrapper
from kolega_code.llm.ledger import UsageLedger
from kolega_code.llm.models import Message, MessageHistory, TextBlock
from kolega_code.llm.providers import openai as transport

pytestmark = pytest.mark.usefixtures("isolated_cli_env")

MODEL = "glm-5.3-abliterated"
FAKE_KEY = "abl_refuseless_offline_lifecycle_not_a_real_key"


class _Body(httpx.AsyncByteStream):
    def __init__(self, *, block: bool = False, credit_error: bool = False, fail_close: bool = False) -> None:
        self.block = block
        self.credit_error = credit_error
        self.fail_close = fail_close
        self.reading = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        event = {
            "id": "chatcmpl-lifecycle",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": MODEL,
            "choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }
        yield f"data: {json.dumps(event)}\n\n".encode()
        if self.block:
            self.reading.set()
            await asyncio.Event().wait()
        if self.credit_error:
            yield b'data: {"error":{"code":"insufficient_credits","message":"Add credits"}}\n\n'
        yield b"data: [DONE]\n\n"

    async def aclose(self) -> None:
        self.closed = True
        if self.fail_close:
            raise RuntimeError("offline close failed")


def _client(
    monkeypatch: pytest.MonkeyPatch, http_client: httpx.AsyncClient
) -> tuple[InstrumentedLLMClient, MagicMock, MagicMock, UsageLedger]:
    monkeypatch.setattr(transport, "AsyncOpenAI", partial(AsyncOpenAI, http_client=http_client))
    langfuse, trace, generation = MagicMock(), MagicMock(), MagicMock()
    langfuse.start_as_current_observation.return_value.__enter__.return_value = trace
    trace.start_observation.return_value = generation
    ledger = UsageLedger()
    client = InstrumentedLLMClient(
        "refuseless",
        FAKE_KEY,
        max_retries=0,
        model=MODEL,
        langfuse_client=langfuse,
        usage_ledger=ledger,
    )
    monkeypatch.setattr(client, "_trace_attributes", lambda _: nullcontext())
    monkeypatch.setattr(client, "_record_usage", AsyncMock())
    return client, generation, trace, ledger


async def _stream(client: InstrumentedLLMClient) -> Any:
    result = client.stream(MessageHistory([Message("user", [TextBlock("hello")])]), model=MODEL)
    assert inspect.isawaitable(result)
    return await result


def _assert_failed_observations(generation: MagicMock, trace: MagicMock) -> None:
    for observation in (generation, trace):
        assert observation.update.call_args.kwargs["level"] == "ERROR"
        assert observation.update.call_args.kwargs["status_message"] != "Success"
        observation.end.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_close", [False, True])
async def test_blocked_sdk_read_cancellation_closes_and_finalizes_partial_usage(
    monkeypatch: pytest.MonkeyPatch, fail_close: bool
) -> None:
    body = _Body(block=True, fail_close=fail_close)
    response = httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as http_client:
        client, generation, trace, ledger = _client(monkeypatch, http_client)
        stream = await _stream(client)
        chunks = []

        async def consume() -> None:
            async with stream:
                async for chunk in stream:
                    chunks.append(chunk)

        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(body.reading.wait(), 10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 10)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        assert [chunk.text for chunk in chunks] == ["partial"]
        assert body.closed
        assert stream.stream._closed
        assert not stream.stream.get_usage_observation().complete
        _assert_failed_observations(generation, trace)
        assert generation.update.call_args.kwargs["usage_details"] == {"input": 10, "output": 2, "total": 12}
        recorder: Any = client._record_usage
        recorder.assert_awaited_once()
        assert recorder.await_args.kwargs["success"] is False
        snapshot = ledger.snapshot()
        assert (snapshot.failed, snapshot.responses) == (1, 0)
        assert (snapshot.input_tokens, snapshot.output_tokens, snapshot.total_tokens) == (10, 2, 12)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["consumer", "credit", "close"])
async def test_failed_sdk_close_preserves_primary_error_and_finalizes(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    body = _Body(credit_error=failure == "credit", fail_close=True)
    response = httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as http_client:
        client, generation, trace, ledger = _client(monkeypatch, http_client)
        stream = await _stream(client)
        primary = ValueError("offline consumer failed")
        expected = ValueError if failure == "consumer" else APIError if failure == "credit" else RuntimeError
        with pytest.raises(expected) as caught:
            async with stream:
                async for _ in stream:
                    if failure == "consumer":
                        raise primary
        if failure == "consumer":
            assert caught.value is primary
        elif failure == "credit":
            assert getattr(caught.value, "code", None) == "insufficient_credits"
        else:
            assert str(caught.value) == "offline close failed"
        assert body.closed
        assert stream.stream._closed
        _assert_failed_observations(generation, trace)
        recorder: Any = client._record_usage
        recorder.assert_awaited_once()
        assert recorder.await_args.kwargs["success"] is False
        assert ledger.snapshot().failed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["credit", "cancel"])
async def test_stream_setup_failure_ends_both_observations(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    started = asyncio.Event()
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        started.set()
        if failure == "cancel":
            await asyncio.Event().wait()
        return httpx.Response(402, json={"error": {"code": "insufficient_credits", "message": "Add credits"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client, generation, trace, ledger = _client(monkeypatch, http_client)
        task = asyncio.create_task(_stream(client))
        try:
            await asyncio.wait_for(started.wait(), 10)
            if failure == "cancel":
                task.cancel()
            with pytest.raises(asyncio.CancelledError if failure == "cancel" else APIStatusError):
                await asyncio.wait_for(task, 10)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        assert len(requests) == 1
        _assert_failed_observations(generation, trace)
        assert ledger.snapshot().failed == 1


@pytest.mark.asyncio
async def test_stream_enter_failure_closes_and_ends_both_observations() -> None:
    primary = ValueError("offline enter failed")
    inner = MagicMock()
    inner.__aenter__ = AsyncMock(side_effect=primary)
    inner.__aexit__ = AsyncMock(side_effect=RuntimeError("offline close failed"))
    inner.get_final_message = AsyncMock(return_value=Message("assistant", [TextBlock("partial")]))
    generation, trace = MagicMock(), MagicMock()
    client = MagicMock(_record_usage=AsyncMock())
    stream = MinimalLangfuseStreamWrapper(inner, generation, trace, client, MODEL)
    with pytest.raises(ValueError) as caught:
        async with stream:
            pytest.fail("enter must fail")
    assert caught.value is primary
    assert await stream.__aexit__(type(primary), primary, primary.__traceback__) is False
    inner.__aexit__.assert_awaited_once()
    _assert_failed_observations(generation, trace)


@pytest.mark.asyncio
@pytest.mark.parametrize("primary_error", [False, True])
async def test_wrapper_close_failure_finalizes_and_only_propagates_without_primary_error(primary_error: bool) -> None:
    primary = ValueError("offline consumer failed") if primary_error else None
    inner = MagicMock()
    inner.__aenter__ = AsyncMock()
    inner.__aexit__ = AsyncMock(side_effect=RuntimeError("offline close failed"))
    inner.get_final_message = AsyncMock(return_value=Message("assistant", [TextBlock("partial")]))
    generation, trace = MagicMock(), MagicMock()
    stream = MinimalLangfuseStreamWrapper(inner, generation, trace, MagicMock(), MODEL)
    with pytest.raises(ValueError if primary_error else RuntimeError) as caught:
        async with stream:
            if primary is not None:
                raise primary
    if primary is not None:
        assert caught.value is primary
    else:
        assert str(caught.value) == "offline close failed"
    _assert_failed_observations(generation, trace)
    assert inner.get_final_message.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("broken_method", ["update", "end"])
async def test_failed_observation_does_not_skip_the_other_observation(broken_method: str) -> None:
    inner = MagicMock()
    inner.__aenter__ = AsyncMock()
    inner.__aexit__ = AsyncMock()
    inner.get_final_message = AsyncMock(return_value=Message("assistant", [TextBlock("partial")]))
    generation, trace = MagicMock(), MagicMock()
    getattr(generation, broken_method).side_effect = RuntimeError("offline tracing failure")
    stream = MinimalLangfuseStreamWrapper(inner, generation, trace, MagicMock(), MODEL)
    primary = ValueError("offline consumer failed")
    with pytest.raises(ValueError) as caught:
        async with stream:
            raise primary
    assert caught.value is primary
    _assert_failed_observations(generation, trace)


@pytest.mark.asyncio
async def test_final_message_cancellation_still_ends_observations() -> None:
    inner = MagicMock()
    inner.__aenter__ = AsyncMock()
    inner.__aexit__ = AsyncMock()
    inner.get_final_message = AsyncMock(side_effect=asyncio.CancelledError())
    generation, trace = MagicMock(), MagicMock()
    stream = MinimalLangfuseStreamWrapper(inner, generation, trace, MagicMock(), MODEL)
    with pytest.raises(asyncio.CancelledError):
        async with stream:
            pass
    _assert_failed_observations(generation, trace)
