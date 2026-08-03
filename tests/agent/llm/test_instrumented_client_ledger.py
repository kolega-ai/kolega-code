"""Ledger settlement through InstrumentedLLMClient — with and without Langfuse.

The Langfuse stream path is the one that historically triple-finalizes
(MinimalLangfuseStreamWrapper.__aexit__ calls get_final_message twice, then the
caller once more): the ledger must record exactly one settlement."""

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from kolega_code.llm.instrumented_client import InstrumentedLLMClient
from kolega_code.llm.ledger import UsageLedger

from .test_client_ledger import _FakeProvider, _messages


def _mock_langfuse():
    langfuse = MagicMock()
    generation = MagicMock()
    trace = MagicMock()
    trace.start_generation = MagicMock(return_value=generation)
    langfuse.start_span = MagicMock(return_value=trace)
    return langfuse


def _instrumented(ledger, provider=None, langfuse=None):
    client = InstrumentedLLMClient(
        provider="anthropic",
        api_key="test-key",
        langfuse_client=langfuse,
        usage_ledger=ledger,
    )
    client.provider = cast(Any, provider or _FakeProvider())
    return client


def test_ctor_threads_ledger_to_base():
    ledger = UsageLedger()
    client = _instrumented(ledger)
    assert client._usage_ledger is ledger


@pytest.mark.parametrize("with_langfuse", [False, True])
@pytest.mark.asyncio
async def test_generate_settles_once(with_langfuse):
    ledger = UsageLedger()
    client = _instrumented(ledger, langfuse=_mock_langfuse() if with_langfuse else None)
    await client.generate(messages=_messages(), model="claude-opus-5")
    snap = ledger.snapshot()
    assert (snap.requests, snap.reported) == (1, 1)
    assert snap.total_tokens == 15


@pytest.mark.asyncio
async def test_generate_error_with_langfuse_settles_failed_once():
    ledger = UsageLedger()
    client = _instrumented(ledger, provider=_FakeProvider(generate_error=ValueError("boom")), langfuse=_mock_langfuse())
    with pytest.raises(Exception):
        await client.generate(messages=_messages(), model="claude-opus-5")
    snap = ledger.snapshot()
    assert (snap.requests, snap.failed) == (1, 1)


@pytest.mark.parametrize("with_langfuse", [False, True])
@pytest.mark.asyncio
async def test_stream_triple_finalization_settles_once(with_langfuse):
    ledger = UsageLedger()
    provider = _FakeProvider()
    client = _instrumented(ledger, provider=provider, langfuse=_mock_langfuse() if with_langfuse else None)

    stream_cm = await cast(Any, client.stream(messages=_messages(), model="claude-opus-5"))
    async with stream_cm as stream:
        async for _ in stream:
            pass
    # BaseAgent finalizes after __aexit__; with Langfuse, __aexit__ itself has
    # already called get_final_message twice by now.
    message = await stream.get_final_message()

    assert message is provider.wrapper.message
    if with_langfuse:
        assert provider.wrapper.final_calls == 3
    snap = ledger.snapshot()
    assert (snap.requests, snap.responses, snap.reported) == (1, 1, 1)
    assert snap.total_tokens == 15
    assert snap.complete is True
