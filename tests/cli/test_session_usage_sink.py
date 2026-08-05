"""SessionUsageSink: exactly-once journaling of ledger settlements."""

import asyncio
from pathlib import Path

import pytest

from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.session_usage import (
    LLM_MESSAGE_EVENT,
    LLM_REQUEST_FAILED_EVENT,
    LLM_RUN_STARTED_EVENT,
    SessionUsageSink,
)
from kolega_code.llm.ledger import (
    HISTORY_ORIGIN,
    LedgerStreamAdapter,
    LlmCallOrigin,
    UsageLedger,
    helper_origin,
    llm_call_origin,
)
from kolega_code.llm.models import Message, TextBlock, ThinkingBlock
from kolega_code.llm.usage import normalize_usage


def _usage(inp=10, out=5):
    return normalize_usage({"input_tokens": inp, "output_tokens": out}, "anthropic", "claude-opus-5")


def _message(inp=10, out=5, content="ok"):
    return Message(role="assistant", content=content, usage=_usage(inp, out))


async def _harness(tmp_path: Path, *, mode="tui"):
    store = SessionStore(tmp_path / "state")
    session = store.create(tmp_path / "proj", "code", {})
    recorder = store.recorder(session.session_id)
    ledger = UsageLedger()
    sink = SessionUsageSink(store.journal(session.session_id), recorder, ledger, mode=mode)
    ledger.observer = sink
    await sink.start()
    return store, session, recorder, ledger, sink


def _events(store, session):
    return store.journal(session.session_id).read_events()


@pytest.mark.asyncio
async def test_marker_written_synchronously_before_any_event(tmp_path):
    store, session, _recorder, ledger, sink = await _harness(tmp_path)
    types = [e.event_type for e in _events(store, session)]
    assert types[-1] == LLM_RUN_STARTED_EVENT
    marker = _events(store, session)[-1]
    assert marker.payload == {"run_id": ledger.run_id, "mode": "tui"}
    await sink.aclose()


@pytest.mark.asyncio
async def test_settlements_journal_in_fifo_order(tmp_path):
    store, session, _recorder, ledger, sink = await _harness(tmp_path)
    with llm_call_origin(helper_origin("compression")):
        first = ledger.begin("anthropic", "m")
        second = ledger.begin("anthropic", "m")
        third = ledger.begin("anthropic", "m")
    ledger.record_response(first, _usage(), message=_message(content="one"))
    ledger.record_failure(second, "boom")
    ledger.record_response(third, _usage(), message=_message(content="three"))
    await sink.aclose()

    tail = [e for e in _events(store, session) if e.event_type.startswith("llm.")]
    assert [e.event_type for e in tail] == [
        LLM_RUN_STARTED_EVENT,
        LLM_MESSAGE_EVENT,
        LLM_REQUEST_FAILED_EVENT,
        LLM_MESSAGE_EVENT,
    ]
    assert [e.payload.get("request_id") for e in tail[1:]] == [first, second, third]


class _FakeStream:
    def __init__(self, message):
        self.message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def get_final_message(self):
        return self.message


@pytest.mark.asyncio
async def test_triple_finalization_journals_exactly_once(tmp_path):
    store, session, _recorder, ledger, sink = await _harness(tmp_path)
    with llm_call_origin(LlmCallOrigin(kind="sub_agent", agent_name="Investigator", agent_id="a1")):
        request_id = ledger.begin("anthropic", "claude-opus-5")
    adapter = LedgerStreamAdapter(_FakeStream(_message()), ledger, request_id)
    async with adapter as stream:
        async for _ in stream:
            pass
    await stream.get_final_message()
    await stream.get_final_message()
    await stream.get_final_message()
    await sink.aclose()

    messages = [e for e in _events(store, session) if e.event_type == LLM_MESSAGE_EVENT]
    assert len(messages) == 1
    assert messages[0].payload["origin"] == {"kind": "sub_agent", "agent_name": "Investigator", "agent_id": "a1"}
    assert messages[0].payload["message"]["usage"]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_history_response_skipped_but_history_failure_journaled(tmp_path):
    store, session, _recorder, ledger, sink = await _harness(tmp_path)
    with llm_call_origin(HISTORY_ORIGIN):
        ok = ledger.begin("anthropic", "m")
        bad = ledger.begin("anthropic", "m")
    ledger.record_response(ok, _usage(), message=_message())
    ledger.record_failure(bad, "rate limited")
    await sink.aclose()

    tail = [e.event_type for e in _events(store, session) if e.event_type.startswith("llm.")]
    assert LLM_MESSAGE_EVENT not in tail
    failure = next(e for e in _events(store, session) if e.event_type == LLM_REQUEST_FAILED_EVENT)
    assert failure.payload["origin"] == {"kind": "history"}
    assert failure.payload["error"] == "rate limited"


@pytest.mark.asyncio
async def test_message_none_settlement_still_journaled(tmp_path):
    store, session, _recorder, ledger, sink = await _harness(tmp_path)
    with llm_call_origin(helper_origin("web_fetch")):
        request_id = ledger.begin("openai", "gpt-5.4-mini")
    ledger.record_response(request_id, None)
    await sink.aclose()

    event = next(e for e in _events(store, session) if e.event_type == LLM_MESSAGE_EVENT)
    assert event.payload["message"] is None
    assert event.payload["provider"] == "openai"


@pytest.mark.asyncio
async def test_no_origin_journals_as_unknown(tmp_path):
    store, session, _recorder, ledger, sink = await _harness(tmp_path)
    request_id = ledger.begin("anthropic", "m")
    ledger.record_response(request_id, _usage(), message=_message())
    await sink.aclose()

    event = next(e for e in _events(store, session) if e.event_type == LLM_MESSAGE_EVENT)
    assert event.payload["origin"] == {"kind": "unknown"}


@pytest.mark.asyncio
async def test_turn_id_stamped_when_turn_open_and_none_otherwise(tmp_path):
    store, session, recorder, ledger, sink = await _harness(tmp_path)
    with llm_call_origin(helper_origin("compression")):
        outside = ledger.begin("anthropic", "m")
    ledger.record_response(outside, _usage(), message=_message())

    turn_id = recorder.start_turn(Message(role="user", content=[TextBlock(text="hi")]))
    with llm_call_origin(helper_origin("compression")):
        inside = ledger.begin("anthropic", "m")
    ledger.record_response(inside, _usage(), message=_message())
    recorder.record_assistant(_message())
    recorder.finish_turn("completed")
    await sink.aclose()

    by_request = {e.payload.get("request_id"): e for e in _events(store, session) if e.event_type == LLM_MESSAGE_EVENT}
    assert by_request[outside].turn_id is None
    assert by_request[inside].turn_id == turn_id


@pytest.mark.asyncio
async def test_write_failure_counts_and_drainer_continues(tmp_path, monkeypatch):
    store, session, _recorder, ledger, sink = await _harness(tmp_path)
    journal = store.journal(session.session_id)
    original_append = journal.append
    calls = {"n": 0}

    def flaky_append(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        return original_append(*args, **kwargs)

    monkeypatch.setattr(journal, "append", flaky_append)
    with llm_call_origin(helper_origin("compression")):
        first = ledger.begin("anthropic", "m")
        second = ledger.begin("anthropic", "m")
    ledger.record_response(first, _usage(), message=_message())
    ledger.record_response(second, _usage(), message=_message(content="second"))
    await sink.aclose()

    assert ledger.persist_failures == 1
    messages = [e for e in _events(store, session) if e.event_type == LLM_MESSAGE_EVENT]
    assert [e.payload["request_id"] for e in messages] == [second]


@pytest.mark.asyncio
async def test_concurrent_settlements_each_journal_once(tmp_path):
    store, session, _recorder, ledger, sink = await _harness(tmp_path)

    async def one_call(index: int):
        with llm_call_origin(LlmCallOrigin(kind="sub_agent", agent_name=f"agent-{index}", agent_id=str(index))):
            request_id = ledger.begin("anthropic", "m")
        await asyncio.sleep(0)
        ledger.record_response(request_id, _usage(), message=_message())
        return request_id

    request_ids = await asyncio.gather(*(one_call(i) for i in range(8)))
    await sink.aclose()

    messages = [e for e in _events(store, session) if e.event_type == LLM_MESSAGE_EVENT]
    assert sorted(e.payload["request_id"] for e in messages) == sorted(request_ids)
    assert {e.payload["origin"]["agent_id"] for e in messages} == {str(i) for i in range(8)}


@pytest.mark.asyncio
async def test_opaque_provider_fields_externalized_like_assistant_message(tmp_path):
    store, session, _recorder, ledger, sink = await _harness(tmp_path)
    message = Message(
        role="assistant",
        content=[ThinkingBlock(thinking="secret reasoning", signature="opaque-signature"), TextBlock(text="done")],
        usage=_usage(),
    )
    with llm_call_origin(helper_origin("web_fetch")):
        request_id = ledger.begin("anthropic", "m")
    ledger.record_response(request_id, _usage(), message=message)
    await sink.aclose()

    event = next(e for e in _events(store, session) if e.event_type == LLM_MESSAGE_EVENT)
    thinking_block = event.payload["message"]["content"][0]
    assert thinking_block["signature"] == ""  # externalized, not inline
    assert "signature" in thinking_block["artifact_fields"]
    assert event.artifacts and event.artifacts[0]["purpose"] == "provider_signature"
