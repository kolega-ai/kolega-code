"""Retry policy for stalled streams: bounded in attempts *and* in wall clock.

Attempt counts alone are not a budget. Four attempts against a five-minute
deadline is twenty minutes of silence, which reads as a frozen application. A
stall that never produced a single character is treated as an unreachable
endpoint and retried once.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from kolega_code.llm.exceptions import LLMRateLimitError, LLMStreamStalled

#: The strip's label budget on a 120-column terminal; longer text is cut mid-word.
_STRIP_BUDGET = 52


def _zero_output_stall(waited_s: float = 90.0) -> LLMStreamStalled:
    return LLMStreamStalled(
        "No response from the endpoint",
        provider="deepseek",
        stage="first_chunk",
        waited_s=waited_s,
        timeout_s=waited_s,
        bytes_received=0,
        partial_output=False,
    )


def _partial_stall(waited_s: float = 300.0) -> LLMStreamStalled:
    return LLMStreamStalled(
        "The response stalled mid-stream",
        provider="deepseek",
        stage="stream",
        waited_s=waited_s,
        timeout_s=waited_s,
        bytes_received=120,
        partial_output=True,
    )


async def _no_sleep(_delay: float) -> None:
    return None


def _status_events(mock_connection_manager) -> list:
    return [
        call.args[0]
        for call in mock_connection_manager.broadcast_event.await_args_list
        if call.args[0].event_type == "llm_status_update"
    ]


@pytest.fixture
def endpoint(base_agent, monkeypatch):
    """Give the agent a resolvable endpoint so messages can name it."""
    monkeypatch.setattr(base_agent.llm.provider, "base_url", "https://api.example.test", raising=False)
    return "https://api.example.test"


@pytest.mark.asyncio
async def test_zero_output_stall_is_retried_only_once(base_agent):
    with patch("kolega_code.agent.baseagent.asyncio.sleep", side_effect=_no_sleep):
        await base_agent.handle_llm_error(_zero_output_stall())

        with pytest.raises(LLMStreamStalled):
            await base_agent.handle_llm_error(_zero_output_stall())


@pytest.mark.asyncio
async def test_partial_stream_stall_keeps_the_normal_attempt_budget(base_agent):
    cap = base_agent.primary_model_config.rate_limits.loop_max_retries

    with patch("kolega_code.agent.baseagent.asyncio.sleep", side_effect=_no_sleep):
        for _ in range(cap):
            await base_agent.handle_llm_error(_partial_stall())
        with pytest.raises(LLMStreamStalled):
            await base_agent.handle_llm_error(_partial_stall())


@pytest.mark.asyncio
async def test_wall_clock_budget_stops_retries(base_agent):
    """Even a patient attempt cap must not buy unbounded silence."""
    base_agent._llm_retry_budget_seconds = 0.0

    with (
        patch("kolega_code.agent.baseagent.asyncio.sleep", side_effect=_no_sleep),
        pytest.raises(LLMStreamStalled),
    ):
        await base_agent.handle_llm_error(_partial_stall())


@pytest.mark.asyncio
async def test_retry_status_carries_structure_for_the_transcript(base_agent, mock_connection_manager, endpoint):
    with patch("kolega_code.agent.baseagent.asyncio.sleep", side_effect=_no_sleep):
        await base_agent.handle_llm_error(_partial_stall())

    events = _status_events(mock_connection_manager)
    assert len(events) == 1
    content = events[0].content
    assert content["retry"] is True
    assert content["attempt"] == 1
    assert content["cap"] == base_agent.primary_model_config.rate_limits.loop_max_retries
    assert content["stalled"] is True
    assert content["partial_output"] is True
    message = content["message"]
    assert message.startswith("Connection issue — retrying (1/")
    assert len(message) <= _STRIP_BUDGET
    assert endpoint not in message
    assert "stalled" not in message.lower()


@pytest.mark.asyncio
async def test_rate_limit_retry_uses_the_same_short_notice(base_agent, mock_connection_manager):
    body = "OpenAI APIError: Error code: 429 - Rate limit reached for gpt-5 on requests per min (RPM): Limit 500."

    with patch("kolega_code.agent.baseagent.asyncio.sleep", side_effect=_no_sleep):
        await base_agent.handle_llm_error(LLMRateLimitError(message=body, provider="openai"))

    message = _status_events(mock_connection_manager)[0].content["message"]
    assert message.startswith("Connection issue — retrying (1/")
    assert len(message) <= _STRIP_BUDGET
    assert "429" not in message


@pytest.mark.asyncio
async def test_stall_heartbeat_fits_the_status_strip(base_agent, mock_connection_manager):
    await base_agent._emit_stream_wait_status(0, waited_s=75.0, limit_s=90.0)

    content = _status_events(mock_connection_manager)[0].content
    assert content["message"] == "Connection issue — waiting for a response…"
    assert len(content["message"]) <= _STRIP_BUDGET
    assert content["seconds_since_data"] == 75.0
    assert content["timeout_s"] == 90.0


@pytest.mark.asyncio
async def test_exhausted_message_counts_attempts_without_the_endpoint(base_agent, mock_connection_manager, endpoint):
    base_agent._llm_retry_budget_seconds = 0.0

    with (
        patch("kolega_code.agent.baseagent.asyncio.sleep", side_effect=_no_sleep),
        pytest.raises(LLMStreamStalled),
    ):
        await base_agent.handle_llm_error(_zero_output_stall(waited_s=91.0))

    events = [event for event in _status_events(mock_connection_manager) if event.content["status"] == "error"]
    assert events, "no terminal error status was emitted"
    message = events[-1].content["message"]
    assert message.startswith("Connection issue — no response after 1 attempt")
    assert endpoint not in message
    assert "VPN" not in message
