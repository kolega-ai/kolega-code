"""Client-layer settlement tests: every LLMClient invocation settles into the
ledger exactly once, across generate/stream, failures, cancellation, and
repeated stream finalization."""

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any, cast

import pytest

from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import LLMError
from kolega_code.llm.ledger import LedgerStreamAdapter, UsageLedger
from kolega_code.llm.models import Message, MessageHistory
from kolega_code.llm.usage import normalize_usage

REASON_NOT_REPORTED = "provider_did_not_report_usage"


def _message(inp=10, out=5):
    usage = normalize_usage({"input_tokens": inp, "output_tokens": out}, "anthropic", "claude-opus-5")
    return Message(role="assistant", content="ok", usage=usage)


class _FakeWrapper:
    def __init__(self, message=None, chunks=1, anext_error=None, final_error=None):
        self.message = message if message is not None else _message()
        self.chunks = chunks
        self.anext_error = anext_error
        self.final_error = final_error
        self.final_calls = 0
        self._emitted = 0
        self.provider_name = "anthropic"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._emitted >= self.chunks:
            raise StopAsyncIteration
        self._emitted += 1
        if self.anext_error is not None:
            raise self.anext_error
        return {"type": "chunk"}

    async def get_final_message(self):
        self.final_calls += 1
        if self.final_error is not None:
            raise self.final_error
        return self.message


class _FakeProvider:
    def __init__(self, message=None, wrapper=None, generate_error=None, stream_error=None):
        self.message = message if message is not None else _message()
        self.wrapper = wrapper if wrapper is not None else _FakeWrapper()
        self.generate_error = generate_error
        self.stream_error = stream_error
        self.generate_calls = 0

    async def generate(self, messages, system=None, params=None, **kwargs):
        self.generate_calls += 1
        if self.generate_error is not None:
            raise self.generate_error
        return self.message

    async def stream(self, messages, system=None, params=None, **kwargs):
        if self.stream_error is not None:
            raise self.stream_error
        return self.wrapper


def _client(ledger, provider=None):
    client = LLMClient(provider="anthropic", api_key="test-key", usage_ledger=ledger)
    client.provider = cast(Any, provider or _FakeProvider())
    return client


async def _open_stream(client, **kwargs):
    return await cast(Any, client.stream(messages=_messages(), **kwargs))


def _messages():
    return MessageHistory([Message(role="user", content="hi")])


# --- generate -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_success_settles_reported():
    ledger = UsageLedger()
    client = _client(ledger)
    message = await client.generate(messages=_messages(), model="claude-opus-5")
    assert message.usage is not None
    snap = ledger.snapshot()
    assert (snap.requests, snap.reported, snap.total_tokens) == (1, 1, 15)
    assert snap.complete is True


@pytest.mark.asyncio
async def test_generate_provider_error_settles_failed_and_maps():
    ledger = UsageLedger()
    client = _client(ledger, _FakeProvider(generate_error=ValueError("boom")))
    with pytest.raises(LLMError):
        await client.generate(messages=_messages())
    snap = ledger.snapshot()
    assert (snap.requests, snap.failed) == (1, 1)
    assert snap.complete is False


@pytest.mark.asyncio
async def test_generate_cancellation_settles_failed_and_propagates_raw():
    ledger = UsageLedger()
    client = _client(ledger, _FakeProvider(generate_error=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await client.generate(messages=_messages())
    assert ledger.snapshot().failed == 1


@pytest.mark.asyncio
async def test_generate_message_without_usage_is_unreported():
    ledger = UsageLedger()
    client = _client(ledger, _FakeProvider(message=SimpleNamespace(content="ok")))
    await client.generate(messages=_messages())
    snap = ledger.snapshot()
    assert snap.unreported == 1
    assert snap.total_tokens == 0
    assert snap.complete is False


@pytest.mark.asyncio
async def test_generate_fan_out_on_one_client():
    ledger = UsageLedger()
    client = _client(ledger)
    await asyncio.gather(*(client.generate(messages=_messages()) for _ in range(10)))
    snap = ledger.snapshot()
    assert (snap.requests, snap.reported, snap.total_tokens) == (10, 10, 150)


@pytest.mark.asyncio
async def test_poisoned_ledger_never_breaks_or_retries_the_request():
    ledger = UsageLedger()

    class _Poison:
        def __setitem__(self, key, value):
            raise RuntimeError("poisoned")

        def get(self, key):
            raise RuntimeError("poisoned")

    cast(Any, ledger)._records = _Poison()
    provider = _FakeProvider()
    client = _client(ledger, provider)
    message = await client.generate(messages=_messages())
    assert message is provider.message
    assert provider.generate_calls == 1


# --- ledger disabled ------------------------------------------------------------


@pytest.mark.asyncio
async def test_without_ledger_stream_is_identity_unwrapped():
    client = LLMClient(provider="anthropic", api_key="test-key")
    provider = _FakeProvider()
    client.provider = cast(Any, provider)
    result = client.stream(messages=_messages())
    assert inspect.iscoroutine(result)
    wrapper = await result
    assert wrapper is provider.wrapper


# --- stream ---------------------------------------------------------------------


async def _drive(client, *, finalize="after"):
    """Run a stream to completion; finalize inside or after the async with."""
    stream_cm = await _open_stream(client, model="claude-opus-5")
    assert isinstance(stream_cm, LedgerStreamAdapter)
    async with stream_cm as stream:
        async for _ in stream:
            pass
        if finalize == "inside":
            message = await stream.get_final_message()
    if finalize == "after":
        message = await stream.get_final_message()
    return stream_cm, message


@pytest.mark.parametrize("finalize", ["after", "inside"])
@pytest.mark.asyncio
async def test_stream_settles_once_regardless_of_finalize_position(finalize):
    ledger = UsageLedger()
    client = _client(ledger)
    adapter, message = await _drive(client, finalize=finalize)
    assert message.usage is not None
    snap = ledger.snapshot()
    assert (snap.requests, snap.reported) == (1, 1)

    # Repeated finalization (the Langfuse wrapper calls it up to 3x): no-ops.
    await adapter.get_final_message()
    await adapter.get_final_message()
    snap = ledger.snapshot()
    assert (snap.requests, snap.responses, snap.reported) == (1, 1, 1)
    assert snap.total_tokens == 15


@pytest.mark.asyncio
async def test_stream_exited_without_finalize_stays_open():
    ledger = UsageLedger()
    client = _client(ledger)
    stream_cm = await _open_stream(client)
    async with stream_cm as stream:
        async for _ in stream:
            pass
    snap = ledger.snapshot()
    assert snap.open == 1
    assert snap.complete is False


@pytest.mark.asyncio
async def test_stream_anext_error_settles_failed_once():
    ledger = UsageLedger()
    wrapper = _FakeWrapper(anext_error=RuntimeError("mid-stream"))
    client = _client(ledger, _FakeProvider(wrapper=wrapper))
    stream_cm = await _open_stream(client)
    with pytest.raises(RuntimeError):
        async with stream_cm as stream:
            async for _ in stream:
                pass
    snap = ledger.snapshot()
    assert (snap.requests, snap.failed) == (1, 1)

    # A later finalize attempt cannot flip the settled state.
    message = await stream_cm.get_final_message()
    assert message is wrapper.message
    assert ledger.snapshot().failed == 1
    assert ledger.snapshot().responses == 0


@pytest.mark.asyncio
async def test_stream_cancellation_through_with_block_settles_failed():
    ledger = UsageLedger()
    client = _client(ledger)
    stream_cm = await _open_stream(client)
    with pytest.raises(asyncio.CancelledError):
        async with stream_cm:
            raise asyncio.CancelledError()
    assert ledger.snapshot().failed == 1


@pytest.mark.asyncio
async def test_stream_setup_failure_settles_failed_and_propagates_raw():
    ledger = UsageLedger()
    client = _client(ledger, _FakeProvider(stream_error=RuntimeError("connect failed")))
    with pytest.raises(RuntimeError):
        await _open_stream(client)
    snap = ledger.snapshot()
    assert (snap.requests, snap.failed) == (1, 1)


@pytest.mark.asyncio
async def test_stream_finalize_failure_settles_failed_once():
    ledger = UsageLedger()
    wrapper = _FakeWrapper(final_error=AssertionError("dead stream"))
    client = _client(ledger, _FakeProvider(wrapper=wrapper))
    stream_cm = await _open_stream(client)
    async with stream_cm as stream:
        async for _ in stream:
            pass
    with pytest.raises(AssertionError):
        await stream_cm.get_final_message()
    with pytest.raises(AssertionError):
        await stream_cm.get_final_message()
    snap = ledger.snapshot()
    assert (snap.requests, snap.failed) == (1, 1)


@pytest.mark.asyncio
async def test_adapter_passes_through_wrapper_attributes():
    ledger = UsageLedger()
    wrapper = _FakeWrapper()
    client = _client(ledger, _FakeProvider(wrapper=wrapper))
    stream_cm = await _open_stream(client)
    assert stream_cm.provider_name == "anthropic"
    assert hasattr(stream_cm, "get_final_message")
